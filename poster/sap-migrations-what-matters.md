# SAP Migrations: What Actually Matters

SAP runs 77% of the world's business transactions. If you work with enterprise customers, you will encounter SAP. Understanding what matters in these migrations will help you win deals and serve customers better.

## Why SAP is Strategic

SAP sits at the center of most large enterprises. It handles sales, inventory, manufacturing, financials, and reporting. When a company says they run SAP, they mean their entire revenue operation depends on it.

This is why SAP migrations are strategic. When the core ERP moves to cloud, everything else follows. Competitors know this. They will put unprecedented money on the table to win these deals. They will lose money for three or four years just to get the land grab.

You need to understand this dynamic. SAP is not just another workload. It is the anchor that brings the rest of the portfolio.

## The Complexity is Real

SAP implementations take time. A typical project runs 12 to 18 months minimum. Some run longer. Phillips 66 started their S/4HANA transformation in 2017. They are still working on it.

This is not because people are slow. It is because SAP touches everything. Any regulatory change requires updates. Any business reorganization requires updates. Any merger or acquisition requires integration work.

The application is monolithic. Changes cascade through the system. Testing takes time. Validation takes time. Getting it wrong has real consequences because this is where the financial data lives.

## The Security Conversation

Security is at the core of every SAP conversation. The entire corporate financial record sits in the ERP. A ransomware attack or data breach would be catastrophic.

This is especially true for energy companies and utilities. They are targets. They know it. They care deeply about security controls.

When you talk to these customers, lead with security. Talk about AWS Nitro. Talk about encryption. Talk about compliance. Talk about how we protect mission-critical workloads. This is not a checkbox. This is the foundation of trust.

## The Ecosystem Migration

When you migrate SAP, you do not just migrate SAP. You migrate the ecosystem. Most enterprises have hundreds of systems that integrate with SAP. Some have 500 or more integrations.

You cannot move everything at once. You prioritize. Real-time integrations move first. Systems that affect transactional performance move first. Systems that impact month-end close move first.

But eventually, everything follows. The batch systems. The reporting systems. The analytics systems. The planning systems. All of it migrates because SAP is the core.

This is why SAP migrations grow. You start with the ERP. Then you add the financial consolidation platform. Then you add the supply chain systems. Then you add the manufacturing systems. The footprint expands over time.

## The Key Differentiators

When you compete on SAP, three things matter most.

First, high memory instances. SAP HANA is an in-memory database. Large systems need 12 terabytes, 18 terabytes, or 24 terabytes of memory. AWS has cloud-native high memory instances. Azure and GCP use co-located hardware in most cases. This is a real advantage.

Second, AWS Nitro. When you provision an 8 vCPU, 64 GB instance, the full 8 vCPU and 64 GB are available to SAP. There is no hypervisor overhead. Competitors cannot say this. Nitro also provides security isolation that matters for regulated industries.

Third, partnership depth. AWS and SAP have worked together since 2008. We co-innovate. SAP launches new solutions on AWS first. In many cases, those solutions stay AWS-only for two or three years before porting to other clouds. This gives customers access to innovation earlier.

## The Pricing Reality

Pricing is competitive. All hyperscalers come in at roughly the same price for most workloads. The big differences show up in high memory instances. That is where AWS has an advantage.

Do not lead with price. Lead with outcomes. Lead with security. Lead with innovation. Lead with the partnership. Price will matter, but it should not be the first conversation.

## The Architecture Patterns

Most SAP systems follow one of three patterns.

Single instance architecture is for development, sandbox, and proof of concept systems. One EC2 instance runs the entire application. If that instance goes down, the application is unavailable. This is common in on-prem environments. In cloud, you should push customers toward distributed architectures.

Distributed single-AZ architecture separates components. The database runs on one instance. The primary application server runs on another instance. Additional application servers run on separate instances. This is for training systems, QA systems, and production systems that can tolerate downtime.

High availability multi-AZ architecture distributes components across availability zones. The database has a standby in another AZ. Application servers run in multiple AZs. This is for production systems with strict RTO and RPO requirements. Most large enterprises need this for their core ERP.

## The RTO and RPO Conversation

Every SAP customer has RTO and RPO requirements. RTO is recovery time objective. How long can the system be down? RPO is recovery point objective. How much data can you afford to lose?

Some customers say zero hours RTO and zero RPO. They need continuous availability. Some customers say four hours RTO and zero RPO. They can tolerate downtime but cannot lose transactions. Some customers say one day RTO and four hours RPO. They can tolerate more disruption.

The architecture and backup strategy depend on these requirements. Ask about RTO and RPO early. Design the solution to meet those requirements. Do not assume.

## The Storage and Backup Strategy

EBS is the primary storage for SAP on AWS. Most customers use GP3. Some customers with high IOPS requirements use io2 or io2 Block Express.

For SAP HANA, you can back up directly to S3. For other databases like Oracle or SQL Server, you back up to local disk and then move to S3 using a script or AWS Systems Manager.

Larger systems need more sophisticated backup strategies. Snapshots. Replication. Cross-region copies. The approach scales with the size and criticality of the system.

## The Launch Wizard Value

Setting up a new SAP system on-prem takes one to three months. You order hardware. You install the operating system. You install the database. You configure everything. You install SAP. You tune it.

AWS Launch Wizard automates this. For a single instance system, you can go from zero to a running SAP system in less than an hour. For a high availability cluster, you can do it in less than two hours.

This matters for project systems. Every enterprise runs multiple SAP projects per year. They need temporary systems for testing and validation. Launch Wizard makes this fast and repeatable.

## The Well-Architected Review

The Well-Architected Framework for SAP is a proven set of best practices across five pillars: operational excellence, security, reliability, performance efficiency, and cost optimization.

Use this with customers before they go live. Walk through the review. Identify gaps. Fix them before production.

Nextera did a Well-Architected review. They had all the building blocks for automated high availability but had not implemented it. After the review, they implemented it. Phillips 66 did a review with a partner to evaluate their operations, not just technology. They found ways to improve.

This builds confidence. It shows you care about their success. It reduces risk.

## The US Navy Reference

The US Navy migrated one of the largest SAP ERP systems to AWS. 72,000 users. $7 billion worth of parts and goods moving through the system.

They achieved increased data protection. They reduced migration times by 50%. They reduced reporting time from hours to half an hour. They moved to highly scalable infrastructure.

Use this reference with federal customers, energy customers, and utilities. Security matters to these industries. The Navy reference shows it can be done.

## The Graviton Clarification

AWS announced Graviton support for SAP at re:Invent. This announcement is specific to SAP cloud solutions, not on-prem solutions.

Graviton is not yet certified to run traditional SAP workloads that customers migrate to AWS. We are working with SAP to get there. But as of today, do not position Graviton for SAP ERP migrations.

Customers hear about 30% cost savings with Graviton. They ask if they can use it for SAP. The answer is not yet. Be clear about this. Do not create false expectations.

## The App Flow Opportunity

Amazon AppFlow for SAP is a low-code solution to extract data from SAP or put data into SAP. This solves a major pain point. Customers complain that integrations take months. Development teams have to write custom code. It is slow and expensive.

AppFlow changes this. You can build integrations faster. You can pull data for analytics. You can feed data from other systems into SAP.

AppFlow for SAP is not fully baked yet. Do not oversell it. But by the end of the year, it will be ready to position in every SAP conversation. This will be a key enabler for data lake and analytics use cases.

## The Partner Ecosystem

SAP migrations involve partners. Systems integrators like Accenture, Deloitte, and Capgemini. SAP-specific partners like Lemongrass and Syntax. Technology partners like Veeam for backup.

Know the partner ecosystem. Know who does what. Know who your customer already works with. Bring the right partners into the conversation early.

Partners can accelerate deals. They can provide implementation capacity. They can provide SAP expertise that your team may not have. Use them.

## The Certification Path

AWS offers an SAP on AWS Specialty certification. Encourage your team to get certified. Encourage your customers to get certified.

This builds credibility. It shows you take SAP seriously. It shows you have invested in the expertise to support these workloads.

## The Bottom Line

SAP migrations are strategic. They are complex. They take time. They require deep expertise. They require partnership. They require a focus on security, performance, and reliability.

Lead with outcomes, not price. Lead with security. Lead with the AWS and SAP partnership. Lead with high memory instances and Nitro. Lead with innovation.

Understand the ecosystem. Understand RTO and RPO. Understand the architecture patterns. Understand the partner landscape.

Do the Well-Architected review. Use the US Navy reference. Use Launch Wizard to show speed. Use AppFlow to show innovation.

Win the SAP migration. The rest of the portfolio will follow.
